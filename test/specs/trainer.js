'use strict';
const path = require('path');
const fastText = require('../../index');

it('fastText trainer', async function () {
  let data = path.resolve(path.join(__dirname, '../data/cooking.train.txt'));
  let model_path = path.resolve(path.join(__dirname, '../data/cooking.model'));
  let model = new fastText.Model();
  let options = {
    input: data,
    output: model_path,
    loss: "softmax",
    dim: 200,
    bucket: 2000000
  }

  var res = await model.train('supervised', options);
  console.log('model info after training:', res)
  assert.equal(res.dim, 200, 'dim')
});

it('fastText quantize', async function () {
  let input = path.resolve(path.join(__dirname, '../data/cooking.train.txt'));
  let output = path.resolve(path.join(__dirname, '../data/cooking.model'));
  let model = new fastText.Model();
  let options = {
    input,
    output,
    epoch: 1,
    qnorm: true,
    qout: true,
    retrain: true,
    cutoff: 1000,
  };

  var res = await model.train('quantize', options);
  console.log(res);
});
