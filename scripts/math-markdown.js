const { escapeHTML } = require('hexo-util');

hexo.extend.filter.register('marked:extensions', function (extensions) {
  extensions.push({
    name: 'blogMathBlock',
    level: 'block',
    start(source) {
      return source.indexOf('$$');
    },
    tokenizer(source) {
      const match = /^ {0,3}\$\$([\s\S]+?)\$\$[ \t]*(?:\n|$)/.exec(source);
      if (match) return { type: 'blogMathBlock', raw: match[0] };
    },
    renderer(token) {
      return `<p>${escapeHTML(token.raw.trim())}</p>\n`;
    }
  });

  extensions.push({
    name: 'blogMathInline',
    level: 'inline',
    start(source) {
      return source.indexOf('$');
    },
    tokenizer(source) {
      const match = /^\$\$(?:\\[\s\S]|[^\\$])+?\$\$|^\$(?!\$)(?:\\[^\n]|[^\\$\n])+?\$(?!\$)/.exec(source);
      if (match) return { type: 'blogMathInline', raw: match[0] };
    },
    renderer(token) {
      return escapeHTML(token.raw);
    }
  });
});
