'use strict';
const path = require('path');
const fastText = require('../../index');

it('fastText model', async function () {
    let model_path = path.resolve(path.join(__dirname, '../models/model_cooking.bin'))
    console.log('File model path: ' + model_path)
    let model = new fastText.Model();

    var info = await model.loadModel(model_path);
    console.log('load model success!!!', info);

    assert.equal(info.model, 'supervised', 'load supervised model');

    var res = await model.predict('Why not put knives in the dishwasher?', 5, 1);

    assert.equal(res.length, 5, 'number of classifications output')
    assert.equal(res[0].label, '__label__knives', 'output is __label__knives');
});
