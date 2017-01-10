document.write('<script src="domain_model.js"></script>')
document.write('<script src="utterance_model.js"></script>')

var ListPanel = new function() {
    var list = null

    this.init = function() {
        list = document.getElementById('list-ul')
    }

    this.onClick = function(element, event) {
        if (element.id == 'list-clear-button') {
            AnnotationTool.clear()
        } else if (element.id == 'list-ul') {
            AnnotationTool.select(event.target.id)
        }
    }

    this.addToFront = function(utterance) {
        var li = document.createElement('li')
        li.id = utterance.getId()
        li.innerText = utterance.getDomain() + '\n' + utterance.getUtterance() + '\n' + utterance.getIob()
        list.insertBefore(li, list.firstChild)
    }

    this.addToBack = function(utterance) {
        var li = document.createElement('li')
        li.id = utterance.getId()
        li.innerText = utterance.getDomain() + '\n' + utterance.getUtterance() + '\n' + utterance.getIob()
        list.appendChild(li)
    }

    this.update = function(utterance) {
        var children = list.children
        for (i in children) {
            if (children[i].id == utterance.getId()) {
                children[i].innerText = utterance.getDomain() + '\n' + utterance.getUtterance() + '\n' + utterance.getIob()
            }
        }
    }

    this.updateAll = function(utterances) {
        while (list.firstChild) list.removeChild(list.firstChild)
        for (i in utterances) this.addToBack(utterances[i])
    }

    this.focus = function(id) {
        console.log('[focus] ' + id)
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

    var addClass = function(element, clsName) {
        if (!element.className.includes(clsName)) {
            element.className += ' ' + clsName
        }
    }

    var removeClass = function(element, clsName) {
        elements.panel.className = elements.panel.className.replace(new RegExp("(\\s|^)" + clsName + "(\\s|$)"), ' ').trim()
    }

    this.init = function(domains) {
        elements.panel = document.getElementById('detail-panel')
        elements.domainSelect = document.getElementById('detail-domain-select')
        elements.idInfo = document.getElementById('detail-id-info')
        elements.sourceInput = document.getElementById('detail-source-input')
        elements.utteranceInfo = document.getElementById('detail-utterance-info')
        elements.iobInfo = document.getElementById('detail-iob-info')

        for (i in domains) {
            elements.domainSelect.appendChild(domains[i].createOption())
        }
    }

    this.onClick = function() {
        var id = this.getId()
        var domain = this.getDomain()
        var source = this.getSource()
        AnnotationTool.updateUtterance(id, domain, source)
    }

    this.show = function(uttr) {
        removeClass(elements.panel, 'gone')
        elements.domainSelect.value = uttr.getDomain()
        elements.idInfo.innerText = uttr.getId()
        elements.sourceInput.value = uttr.getSource()
        elements.utteranceInfo.innerText = uttr.getUtterance()
        elements.iobInfo.innerText = uttr.getIob()
    }

    this.hide = function() {
        addClass(elements.panel, 'gone')
    }

    this.getId = function() {
        return elements.idInfo.innerText
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
    }

    this.save = function() {
        localStorage.setItem('utterances', JSON.stringify(_utterances))
    }

    this.addUtterance = function(domain, source) {
        var model = new UtteranceModel()
        model.setDomain(domain)
        model.setSource(source)
        _utterances.unshift(model)
        ListPanel.addToFront(model)
        this.select(model.getId())
    }

    this.updateUtterance = function(id, domain, source) {
        var uttr = findUtterance(_utterances, id)
        if (uttr != null) {
            uttr.setDomain(domain)
            uttr.setSource(source)
            ListPanel.update(uttr)
            this.select(id)
        }
    }

    this.select = function(id) {
        var uttr = findUtterance(_utterances, id)
        if (uttr != null) {
            DetailPanel.show(uttr)
            ListPanel.focus(id)
        }
    }

    this.clear = function() {
        _utterances = []
        ListPanel.updateAll()
        DetailPanel.hide()
    }

    var findUtterance = function(utterances, id) {
        for (i in utterances) {
            if (id == utterances[i].getId()) {
                return utterances[i]
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
