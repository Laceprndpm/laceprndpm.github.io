const { Parser } = require('htmlparser2');
const { escapeHTML } = require('hexo-util');

function rewritePostLinks(html, source, documentUrls) {
  const replacements = [];
  const sourcePath = source.split('/').map(encodeURIComponent).join('/');
  const parser = new Parser({
    onopentag(name, attributes) {
      if (name !== 'a' || !attributes.href) return;
      const href = attributes.href.trim();
      if (/^(?:[a-z][a-z\d+.-]*:|[\/#?])/i.test(href)) return;
      const address = new URL(href, `https://notes.invalid/${sourcePath}`);
      const target = decodeURIComponent(address.pathname).slice(1);
      if (!/\.(?:md|markdown)$/i.test(target)) return;
      const url = documentUrls.get(target);
      if (!url) {
        throw new Error(`Unpublished or missing article: ${source} -> ${href}`);
      }
      attributes.href = url + address.search + address.hash;
      const tag = '<a' + Object.entries(attributes)
        .map(([attribute, value]) => ` ${attribute}="${escapeHTML(value)}"`).join('') + '>';
      replacements.push({ start: parser.startIndex, end: parser.endIndex + 1, tag });
    }
  }, { decodeEntities: true });
  parser.end(html);
  for (const { start, end, tag } of replacements.reverse()) {
    html = html.slice(0, start) + tag + html.slice(end);
  }
  return html;
}

function inspectHtml(html) {
  const ids = new Set();
  const references = [];
  const errors = [];
  let hasHtml = false;
  let hasBody = false;
  const parser = new Parser({
    onopentag(name, attributes) {
      if (name === 'html') hasHtml = true;
      if (name === 'body') hasBody = true;
      if (attributes.id) ids.add(attributes.id);
      if (name === 'a' && attributes.name) ids.add(attributes.name);
      if (attributes['data-mjx-error']) errors.push(attributes['data-mjx-error']);
      const add = (value, kind) => {
        if (value) references.push({ href: value, kind });
      };
      if (['a', 'area'].includes(name)) add(attributes.href, 'link');
      if (name === 'link') add(attributes.href, 'resource');
      if (name === 'img') add(attributes['data-src'] || attributes.src, 'resource');
      if (['script', 'iframe', 'audio', 'video', 'source', 'track'].includes(name)) {
        add(attributes.src, 'resource');
      }
      if (name === 'video') add(attributes.poster, 'resource');
      if (['img', 'source'].includes(name) && attributes.srcset && !attributes.srcset.includes('data:')) {
        for (const candidate of attributes.srcset.split(',')) {
          add(candidate.trim().split(/\s+/)[0], 'resource');
        }
      }
    }
  }, { decodeEntities: true });
  parser.end(html);
  return { ids, references, errors, isDocument: hasHtml && hasBody };
}

function isPrivatePath(value) {
  return /^(?:file:|vscode:|obsidian:|[a-z]:[\\/]|\/(?:home|Users|tmp|private|mnt|Volumes|etc|var)\/)/i.test(value);
}

module.exports = { rewritePostLinks, inspectHtml, isPrivatePath };
