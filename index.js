const path = require('path');
module.exports = require(`./addon/${path.basename(__dirname)}.node`);