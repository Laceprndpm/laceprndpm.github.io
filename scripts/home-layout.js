const { readFileSync } = require('node:fs');
const { join } = require('node:path');

hexo.extend.filter.register('before_generate', function () {
  this.theme.setView('_page/keep-home.ejs', readFileSync(
    join(this.theme_dir, 'layout/_page/home.ejs'), 'utf8'
  ));
  this.theme.setView('_page/home.ejs', readFileSync(
    join(this.base_dir, 'layout/home.ejs'), 'utf8'
  ));
});
