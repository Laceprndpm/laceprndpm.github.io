const fs = require('node:fs');
const path = require('node:path');
const { parseArgs } = require('node:util');
const yaml = require('js-yaml');
const { inspectHtml, isPrivatePath } = require('./site-links.cjs');

function checkSite(publicDirectory, siteAddress) {
  const root = fs.realpathSync(publicDirectory);
  const site = new URL(siteAddress.replace(/\/?$/, '/'));
  const sitePrefix = decodeURIComponent(site.pathname);
  const pages = new Map();
  const externalUrls = new Set();
  const failures = new Set();
  let internalReferences = 0;
  const withinRoot = candidate => candidate === root || candidate.startsWith(root + path.sep);

  for (const relative of fs.readdirSync(root, { recursive: true })) {
    if (/\.html?$/i.test(relative) && fs.statSync(path.join(root, relative)).isFile()) {
      if (!withinRoot(fs.realpathSync(path.join(root, relative)))) {
        failures.add(`${relative}: page points outside public directory`);
        continue;
      }
      pages.set(path.join(root, relative), inspectHtml(fs.readFileSync(path.join(root, relative), 'utf8')));
    }
  }
  if (!pages.get(path.join(root, 'index.html'))?.isDocument) {
    failures.add('index.html: missing or incomplete homepage; run a successful build first.');
  }
  for (const [filename, page] of pages) {
    const relative = path.relative(root, filename).split(path.sep).join('/');
    const pageUrl = new URL(relative.split('/').map(encodeURIComponent).join('/'), site);
    for (const error of page.errors) failures.add(`${relative}: MathJax: ${error}`);
    for (const { href, kind } of page.references) {
      const fail = reason => failures.add(`${relative}: ${reason}: ${href}`);
      if (isPrivatePath(href.trim())) {
        fail('local-machine path cannot be published');
        continue;
      }
      let address;
      try {
        address = new URL(href, pageUrl);
      } catch {
        fail('invalid URL');
        continue;
      }
      if (!['https:', 'http:'].includes(address.protocol)) continue;
      if (address.host !== site.host) {
        address.hash = '';
        externalUrls.add(address.href);
        continue;
      }
      internalReferences += 1;
      let pathname;
      let fragment;
      try {
        pathname = decodeURIComponent(address.pathname);
        fragment = decodeURIComponent(address.hash.slice(1)).split(':~:text=')[0];
      } catch {
        fail('invalid URL encoding');
        continue;
      }
      if (isPrivatePath(pathname)) {
        fail('local-machine path cannot be published');
        continue;
      }
      if (kind === 'link' && /\.(?:md|markdown)$/i.test(pathname)) {
        fail('unresolved Markdown article link');
        continue;
      }
      if (pathname === sitePrefix.slice(0, -1)) pathname += '/';
      if (!pathname.startsWith(sitePrefix)) {
        fail('outside the configured site root');
        continue;
      }
      let target = path.resolve(root, pathname.slice(sitePrefix.length));
      if (!withinRoot(target)) {
        fail('path escapes public directory');
        continue;
      }
      if (fs.existsSync(target) && fs.statSync(target).isDirectory()) target = path.join(target, 'index.html');
      if (!fs.existsSync(target) || !fs.statSync(target).isFile()) {
        fail('target file is missing');
        continue;
      }
      if (!withinRoot(fs.realpathSync(target))) {
        fail('target points outside public directory');
        continue;
      }
      if (fragment && fragment.toLowerCase() !== 'top' && pages.has(target) && !pages.get(target).ids.has(fragment)) {
        fail('anchor is missing');
      }
    }
  }
  return { pages: pages.size, internalReferences, externalUrls, failures: [...failures] };
}

async function checkExternal(urls) {
  const pending = [...urls];
  const warnings = [];
  await Promise.all(Array.from({ length: Math.min(6, pending.length) }, async () => {
    while (pending.length) {
      const url = pending.pop();
      try {
        const response = await fetch(url, {
          method: 'HEAD', redirect: 'manual', signal: AbortSignal.timeout(8000)
        });
        if (response.status >= 300) {
          warnings.push(`${url}: HTTP ${response.status}${response.status < 400 ? ' (redirect not followed)' : ''}`);
        }
      } catch (error) {
        warnings.push(`${url}: ${error.message}`);
      }
    }
  }));
  return warnings.sort();
}

async function main() {
  const { values } = parseArgs({ options: {
    external: { type: 'boolean', default: false },
    'public-dir': { type: 'string' },
    'site-url': { type: 'string' }
  } });
  const project = path.resolve(__dirname, '..');
  const config = yaml.load(fs.readFileSync(path.join(project, '_config.yml'), 'utf8'));
  const report = checkSite(
    path.resolve(project, values['public-dir'] || config.public_dir || 'public'),
    values['site-url'] || config.url
  );
  for (const failure of report.failures) console.error(`ERROR ${failure}`);
  console.log(`Checked ${report.pages} pages and ${report.internalReferences} internal references; ${report.failures.length} errors.`);
  if (values.external) {
    const warnings = await checkExternal(report.externalUrls);
    for (const warning of warnings) console.warn(`WARN ${warning}`);
    console.log(`Checked ${report.externalUrls.size} external URLs; ${warnings.length} warnings (non-blocking).`);
  } else {
    console.log(`External URLs not requested (${report.externalUrls.size}); use npm run check -- --external to check them.`);
  }
  process.exitCode = report.failures.length ? 1 : 0;
}

module.exports = { checkSite, checkExternal };
if (require.main === module) main().catch(error => {
  console.error(`Link check failed: ${error.message}`);
  process.exitCode = 1;
});
