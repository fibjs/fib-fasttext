const path = require('path');
const fastText = require('../../index');

it('fastText get sentence vector', async function () {
    let model_path = path.resolve(path.join(__dirname, '../models/lid.176.ftz'))
    console.log('File model path: ' + model_path)
    let model = new fastText.Model();

    var info = await model.loadModel(model_path);
    console.log('load model success!!!', info);

    var res = await model.get_sentence_vector('vntk');
    assert.equal(res.length, 16);
});
