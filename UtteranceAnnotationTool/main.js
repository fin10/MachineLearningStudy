document.write('<script src="domain_model.js"></script>')
document.write('<script src="utterance_model.js"></script>')

var ListPanel = new function() {
    var elements = {}

    this.init = function() {
        elements.list = document.getElementById('list-ul')
        elements.count = document.getElementById('list-count')
        elements.import = document.getElementById('list-import')
    }

    this.onClick = function(element, event) {
        if (element.id == 'list-clear-button') {
            AnnotationTool.clear()
        } else if (element.id == 'list-import-button') {
            elements.import.click()
        } else if (element.id == 'list-export-button') {
            AnnotationTool.export()
        } else if (element.id == 'list-ul') {
            for (i in event.path) {
                if (event.path[i].localName == 'li') {
                    AnnotationTool.select(event.path[i].id)
                    break;
                }
            }
        }
    }

    var addClass = function(element, clsName) {
        if (!element.className.includes(clsName)) {
            element.className += ' ' + clsName
            element.className = element.className.trim()
        }
    }

    var removeClass = function(element, clsName) {
        if(element != null) {
            element.className = element.className.replace(new RegExp("(\\s|^)" + clsName + "(\\s|$)"), ' ').trim()
        }
    }

    var createListItem = function(model) {
        var li = document.createElement('li')
        li.id = model.getId()

        var domain = document.createElement('div')
        domain.className = 'domain-info round-box'
        domain.innerText = model.getDomain()
        li.appendChild(domain)

        var uttr = document.createElement('div')
        uttr.className = 'info'
        uttr.innerText = model.getUtterance()
        li.appendChild(uttr)

        var iob = document.createElement('iob')
        iob.className = 'info'
        iob.innerText = model.getIob()
        li.appendChild(iob)

        return li
    }

    var addToBack = function(utterance) {
        elements.list.appendChild(createListItem(utterance))
    }

    var updateListItem = function(li, model) {
        li.children[0].innerText = model.getDomain()
        li.children[1].innerText = model.getUtterance()
        li.children[2].innerText = model.getIob()
    }

    this.setCount = function(count) {
        elements.count.innerText = count
    }

    this.addToFront = function(utterance) {
        elements.list.insertBefore(createListItem(utterance), elements.list.firstChild)
        this.setCount(elements.list.children.length)
    }

    this.update = function(utterance) {
        var children = elements.list.children
        for (i in children) {
            if (children[i].id == utterance.getId()) {
                updateListItem(children[i], utterance)
            }
        }
    }

    this.updateAll = function(utterances) {
        while (elements.list.firstChild) elements.list.removeChild(elements.list.firstChild)
        for (i in utterances) addToBack(utterances[i])
        this.setCount(elements.list.children.length)
    }

    this.focus = function(id) {
        removeClass(elements.list.querySelector('.selected'), 'selected')

        for (i in elements.list.children) {
            if (elements.list.children[i].id == id) {
                addClass(elements.list.children[i], 'selected')
                break
            }
        }
    }
}

var EnrollPanel = new function() {
    var elements = {}

    this.init = function(domains) {
        elements.panel = document.getElementById('enroll-panel')
        elements.domainSelect = document.getElementById('enroll-domain-select')
        elements.sourceInput = document.getElementById('enroll-source-input')

        for (i in domains) {
            elements.domainSelect.appendChild(domains[i].createOption())
        }
    }

    this.onClick = function() {
        AnnotationTool.addUtterance(elements.domainSelect.value, elements.sourceInput.value)
        elements.sourceInput.value = null
    }
}

var DetailPanel = new function() {
    var elements = {}
    var id = -1
    var slots = {}

    var getSelectionStart = function(input) {
        var index = input.selectionStart
    	if (input.createTextRange) {
    		var r = document.selection.createRange().duplicate()
    		r.moveEnd('character', input.value.length)
    		index = (r.text == '') ? input.value.length : input.value.lastIndexOf(r.text)
    	}

        while(input.value[index] == ' ') {
            index += 1
        }

    	return index
    }

    var getSelectionEnd = function(input) {
        var index = input.selectionEnd
    	if (input.createTextRange) {
    		var r = document.selection.createRange().duplicate()
    		r.moveStart('character', -input.value.length)
    		index = r.text.length
    	}

        while(input.value[index - 1] == ' ') {
            index -= 1
        }

    	return index
    }

    this.init = function(domains) {
        elements.panel = document.getElementById('detail-panel')
        elements.domainSelect = document.getElementById('detail-domain-select')
        elements.sourceInput = document.getElementById('detail-source-input')
        elements.utteranceInfo = document.getElementById('detail-utterance-info')
        elements.iobInfo = document.getElementById('detail-iob-info')

        for (i in domains) {
            elements.domainSelect.appendChild(domains[i].createOption())
            slots[domains[i].getName()] = domains[i].getSlots()
        }

        $.contextMenu({
            selector: '.context-menu-one',
            callback: function(key, options) {
    			var input = options.$trigger[0];
    			var startIdx = getSelectionStart(input)
    			var endIdx = getSelectionEnd(input)
                var head = input.value.substring(0, startIdx)
                var tail = input.value.substring(endIdx)
                input.value = head + '(' + input.value.substring(startIdx, endIdx) + ')[' + key + ']' + tail
            },
            build: function() {
                return {
                    'items': slots[elements.domainSelect.value]
                }
            }
        })
    }

    this.onClick = function() {
        if (id != -1) {
            var domain = this.getDomain()
            var source = this.getSource()
            var updated = AnnotationTool.updateUtterance(id, domain, source)
            this.show(updated)
        }
    }

    this.show = function(uttr) {
        id = uttr.getId();
        elements.domainSelect.value = uttr.getDomain()
        elements.sourceInput.value = uttr.getSource()
        elements.utteranceInfo.innerText = uttr.getUtterance()
        elements.iobInfo.innerText = uttr.getIob()
    }

    this.getId = function() {
        return id
    }

    this.getDomain = function() {
        return elements.domainSelect.value
    }

    this.getSource = function() {
        return elements.sourceInput.value
    }
}

var AnnotationTool = new function() {
    var _domains = []
    var _utterances = []

    this.init = function() {
         _domains = DomainModel.getModels()
        var uttrs = localStorage.getItem('utterances')
        if (uttrs != null) _utterances = UtteranceModel.parseModels(JSON.parse(uttrs))
        console.log('domain:' + _domains.length + ', utterance:' + _utterances.length)

        EnrollPanel.init(_domains)
        DetailPanel.init(_domains)
        ListPanel.init()
        ListPanel.updateAll(_utterances)
        if (_utterances.length > 0) {
            this.select(_utterances[0].getId())
        }
    }

    this.save = function() {
        localStorage.setItem('utterances', JSON.stringify(_utterances))
    }

    this.addUtterance = function(domain, source) {
        var model = new UtteranceModel()
        model.setDomain(domain)
        model.setSource(source)
        if (!exist(model)) {
            _utterances.unshift(model)
            ListPanel.addToFront(model)
            this.select(model.getId())
        }
    }

    this.updateUtterance = function(id, domain, source) {
        var uttr = find(id)
        if (uttr != null) {
            uttr.setDomain(domain)
            uttr.setSource(source)
            ListPanel.update(uttr)
        }

        return uttr
    }

    this.select = function(id) {
        var uttr = find(id)
        if (uttr != null) {
            DetailPanel.onClick()
            DetailPanel.show(uttr)
            ListPanel.focus(id)
        }
    }

    this.clear = function() {
        _utterances = []
        ListPanel.updateAll()
    }

    this.import = function(input) {
        var file = input.files[0]
		var reader = new FileReader();
		reader.onload = function(e) {
    		var utterances = reader.result.split("\n");
    		for (i in utterances) {
    		    var uttr = utterances[i].trim()
    		    if (uttr.length > 0) {
                    var model = new UtteranceModel()
                    model.setDomain(file.name)
                    model.setSource(uttr)
                    if (!exist(model)) {
                        _utterances.push(model)
                    }
    		    }
    		}

            ListPanel.updateAll(_utterances)
		}

		reader.readAsText(file);
    }

    this.export = function() {
    }

    var exist = function(model) {
        var uttr = model.getUtterance()
        for (i in _utterances) {
            if (uttr == _utterances[i].getUtterance()) {
                console.log('[exist] (' + uttr + ') exists.')
                return true
            }
        }

        return false
    }

    var find = function(id) {
        for (i in _utterances) {
            if (id == _utterances[i].getId()) {
                return _utterances[i]
            }
        }

        console.log('[find] ' + id + ' not found.')
        return null
    }
}

function onBodyLoaded() {
    AnnotationTool.init()
}

function onBodyUnloaded() {
    AnnotationTool.save()
}
