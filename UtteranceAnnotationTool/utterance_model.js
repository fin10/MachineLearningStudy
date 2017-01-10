function UtteranceModel(data) {
    var _data = (data != null) ? data : {
        'id' : new Date().getTime() + '_' + Math.floor((Math.random() * 10000)),
        'domain' : '',
        'source' : '',
        'uttr' : '',
        'iob' : '',
        'length' : 0
    }

    var parseUtterance = function(source) {
        return source.replace(/\(|\)|\[[^\]]+\]/g, '')
    }

    var parseIob = function(source) {
        var regex = /\(([^\)]+)\)\[([\w]+)\]/g
        for (var group = regex.exec(source); group != null; group = regex.exec(source)) {
            var iob = ''
            var chunks = group[1].split(' ')
            for (i in chunks) {
                var indicator = i == 0 ? '/b-' : '/i-'
                iob += chunks[i] + indicator + group[2] + ' '
            }

            source = source.replace(group[0], iob.trim())
        }

        var result = ''
        var chunks = source.split(' ')
        for (i in chunks) {
            var idx = chunks[i].indexOf('/');
            result += idx > 0 ? chunks[i].substr(idx + 1) + ' ' : 'o '
        }

        return result.trim()
    }

    return {
        setSource : function(source) {
            _data.source = source
            _data.uttr = parseUtterance(source)
            _data.iob = parseIob(source)
            _data.length = _data.iob.split(' ').length
        },
        setDomain : function(domain) {
            _data.domain = domain
        },
        getId : function() {
            return _data.id
        },
        getDomain : function() {
            return _data.domain
        },
        getSource : function() {
            return _data.source
        },
        getUtterance : function() {
            return _data.uttr
        },
        getIob : function() {
            return _data.iob
        },
        toJSON : function() {
            return _data
        }
    }
}

UtteranceModel.parseModels = function(json) {
    var models = []
    for (i in json) {
        models[i] = new UtteranceModel(json[i])
    }

    return models
}