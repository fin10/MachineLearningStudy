function DomainModel(domain) {
    var _name = domain['domain']
    var _slots = domain['slots']

    var slots = {}
    for (i in _slots) {
        slots[_slots[i]] = { name: _slots[i] }
    }

    return {
        getName : function() {
            return _name
        },
        getSlots : function() {
            return slots
        },
        createOption : function() {
            var option = document.createElement('option')
            option.value = _name
            option.innerText = _name
            return option
        }
    }
}

DomainModel.getModels = function() {
    var _domains = [{
        'domain' : 'public_transport',
        'slots' : [
            'from_stop',
            'to_stop',
            'direction',
            'departure_time',
            'departure_time_rel',
            'ampm',
            'vehicle',
            'line',
            'arrival_time',
            'duration',
            'distance',
            'num_transfers',
            'alternative'
        ]
    }, {
        'domain' : 'travel',
        'slots' : [
            'departure'
        ]
    }
    ]

    var models = []
    for (i in _domains) {
        models[i] = new DomainModel(_domains[i])
    }
    return models
}
