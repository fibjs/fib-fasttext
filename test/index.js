var test = require("test");
test.setup();

var path = require('path');
var dir = '../test/specs/';

describe('fastText', function () {
    [
        'fastText',
        'word_vector',
        'sentence_vector',
        'langid',
        'trainer',
    ].forEach((script) => {
        require(path.join(dir, script));
    });
});

test.run();
