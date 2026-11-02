const { posix } = require('node:path');
const { url_for: urlFor } = require('hexo-util');
const { rewritePostLinks } = require('../tools/site-links.cjs');

let documentUrls = new Map();

hexo.extend.filter.register('before_generate', function () {
  documentUrls = new Map();
  for (const collection of ['posts', 'pages']) {
    this.locals.get(collection).forEach(document => {
      documentUrls.set(
        posix.normalize(document.source.replace(/\\/g, '/')),
        urlFor.call(this, document.path, { relative: false })
      );
    });
  }
}, 20);

hexo.extend.filter.register('template_locals', function (locals) {
  const rewrite = document => {
    if (!document.source) return;
    for (const field of ['content', 'excerpt', 'more']) {
      if (typeof document[field] === 'string') {
        document[field] = rewritePostLinks(document[field], document.source, documentUrls);
      }
    }
  };
  rewrite(locals.page);
  if (typeof locals.page.posts?.forEach === 'function') locals.page.posts.forEach(rewrite);
  return locals;
});
