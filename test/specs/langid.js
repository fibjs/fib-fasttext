const path = require('path');
const fastText = require('../../index');

it('fastText language identification', async function () {
  let model_path = path.resolve(path.join(__dirname, '../models/lid.176.ftz'))
  console.log('File model path: ' + model_path)
  let model = new fastText.Model(model_path);

  var res = await model.predict('sử dụng vntk với fastext rất tuyệt?', 5);
  assert.equal(res.length, 5, 'number of classifications output')
  assert.equal(res[0].label, '__label__vi', 'output is __label__vi');
  assert.isTrue(res[0].value > 0.99, 'confidence is 99%');
  console.log('Result: ', res);
});
