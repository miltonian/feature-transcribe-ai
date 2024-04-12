const esprima = require('esprima')
const firstidx = 0+2
const path = process.argv[firstidx]
// console.log(`path: ${path}`)
const resp = esprima.parseScript(path, {tokens: true})
console.log(resp.tokens)

