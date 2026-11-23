const assert = require('node:assert/strict');
const {test} = require('node:test');
const {readFileSync} = require('node:fs');
const vm = require('node:vm');
const source = readFileSync(new URL('../ui/theme.js', `file://${__filename}`), 'utf8');
function page(url) {
  const node = () => ({classes: new Set(), style: {}, classList: {
    toggle(name, enabled) { enabled ? this.owner.classes.add(name) : this.owner.classes.delete(name); },
    remove(name) { this.owner.classes.delete(name); }
  }});
  const html = node(), body = node();
  for (const item of [html, body]) item.classList.owner = item;
  const attributes = {}, meta = {};
  let click;
  const button = {setAttribute(key, value) {attributes[key] = value;}, addEventListener(event, fn) {click = fn;}};
  const window = {
    location: {href: url}, matchMedia: () => ({matches: true}),
    localStorage: {
      getItem() {throw Error('Theme must not read old preferences');},
      setItem() {throw Error('Theme must not persist across fresh launches');}
    },
    history: {state: null, replaceState(state, title, next) {window.location.href = String(next);}}
  };
  const document = {documentElement: html, body,
    querySelectorAll: () => [html, body].filter(item => item.classes.has('dark')),
    querySelector: () => ({setAttribute(key, value) {meta[key] = value;}})
  };
  vm.runInNewContext(source, {element: {querySelector: () => button}, document, window, URL});
  return {html, body, attributes, meta, button, window, click: () => click()};
}
test('toggle respects URL, switches both ways and keeps other URL state', () => {
  const p = page('http://localhost/?__theme=dark&recording=keep#result');
  assert(p.body.classes.has('dark'));
  assert.equal(p.meta.content, '#000000');
  assert.equal(p.attributes['aria-checked'], 'true');
  assert.equal(p.attributes.title, 'Switch to light mode');
  p.click();
  assert(!p.html.classes.has('dark'));
  assert(!p.body.classes.has('dark'));
  assert.equal(p.attributes['aria-checked'], 'false');
  assert(p.window.location.href.includes('recording=keep#result'));
  p.click();
  assert.equal(p.meta.content, '#000000');
});
test('fresh launches stay light despite dark system or old saved preference', () => {
  assert(!page('http://localhost/').body.classes.has('dark'));
  assert(!page('http://localhost/?__theme=invalid').body.classes.has('dark'));
  const p = page('http://localhost/');
  p.click();
  assert(p.body.classes.has('dark'));
});
