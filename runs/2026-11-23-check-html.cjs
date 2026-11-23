'use strict';
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');

class Element {
  constructor(tag) {
    this.tag = tag;
    this.children = [];
    this.attributes = {};
    this.listeners = {};
    this.style = { setProperty() {} };
    this.value = '';
  }
  append(child) {
    this.children.push(child);
    if (this.tag === 'select' && this.children.length === 1) this.value = child.value;
  }
  replaceChildren() { this.children = []; }
  setAttribute(key, value) {
    assert.ok(!String(value).includes('NaN'));
    this.attributes[key] = value;
  }
  addEventListener(name, listener) { this.listeners[name] = listener; }
}

for (const filename of process.argv.slice(2)) {
  const html = fs.readFileSync(filename, 'utf8');
  const tags = new Map([...html.split('<script>')[0].matchAll(/<(\w+)\s+[^>]*?\bid="([\w-]+)"/g)].map(([, tag, id]) => [id, new Element(tag)]));
  const script = html.match(/<script>([\s\S]+)<\/script>/)?.[1];
  assert.ok(script);
  vm.runInNewContext(script, { document: {
    getElementById: id => tags.get(id), createElement: name => new Element(name),
    createElementNS: (_ns, name) => new Element(name),
  } });
  assert.equal(tags.get('loss').children[0].tag, 'svg');
  assert.equal(tags.get('horizons').children[0].tag, 'svg');
  if (!tags.get('growth-panel').hidden) {
    const round = tags.get('round');
    assert.ok(tags.get('grid').children.length);
    round.value = round.max;
    round.listeners.input();
    assert.ok(tags.get('counts').textContent.includes('条在用'));
    const seed = tags.get('seed');
    seed.value = seed.children.at(-1).value;
    seed.listeners.change();
    const method = tags.get('method');
    method.value = method.children.at(-1).value;
    method.listeners.change();
    assert.ok(tags.get('grid').children.length);
  }
  console.log(`Loaded both charts and exercised replay controls: ${filename}`);
}
