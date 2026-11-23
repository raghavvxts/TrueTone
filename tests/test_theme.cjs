const assert = require('node:assert/strict');
const {test} = require('node:test');
const {readFileSync} = require('node:fs');
const vm = require('node:vm');
const source = readFileSync(new URL('../ui/theme.js', `file://${__filename}`), 'utf8');
function page(url, saved, storageBlocked = false) {
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
    location: {href: url}, matchMedia: () => ({matches: false}),
    localStorage: {
      getItem() {if (storageBlocked) throw Error('blocked'); return saved;},
      setItem(key, value) {if (storageBlocked) throw Error('blocked'); saved = value;}
    },
    history: {state: null, replaceState(state, title, next) {window.location.href = String(next);}}
  };
  const document = {documentElement: html, body,
    querySelectorAll: () => [html, body].filter(item => item.classes.has('dark')),
    querySelector: () => ({setAttribute(key, value) {meta[key] = value;}})
  };
  vm.runInNewContext(source, {element: {querySelector: () => button}, document, window, URL});
  return {html, body, attributes, meta, button, window, click: () => click(), saved: () => saved};
}
test('toggle respects URL, switches both ways and keeps other URL state', () => {
  const p = page('http://localhost/?__theme=dark&recording=keep#result', 'light');
  assert(p.body.classes.has('dark'));
  assert.equal(p.meta.content, '#000000');
  assert.equal(p.attributes['aria-label'], 'Switch to light mode');
  p.click();
  assert(!p.html.classes.has('dark'));
  assert(!p.body.classes.has('dark'));
  assert.equal(p.saved(), 'light');
  assert(p.window.location.href.includes('recording=keep#result'));
  p.click();
  assert.equal(p.saved(), 'dark');
  assert.equal(p.meta.content, '#000000');
});
test('saved preference and blocked storage both work', () => {
  assert(page('http://localhost/', 'dark').body.classes.has('dark'));
  const p = page('http://localhost/', null, true);
  p.click();
  assert(p.body.classes.has('dark'));
});
