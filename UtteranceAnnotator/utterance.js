var Slot = (function() {
	var Slot = {};
	
	Slot.parseSlotData = function(input) {
		var result = {};
		var dataList = input.split("\n");
		for (var i = 0; i < dataList.length; ++i ) {
			var text = dataList[i].trim();
			if (text == '') continue;

			result[text] = {
				name: text,
				items: {}
			};

			result[text]['items']['B-' + text] = { name : 'B' };
			result[text]['items']['I-' + text] = { name : 'I' };
		}

		console.log(result);
		return result;
	}
	
	return Slot;
})();

var Intent = (function() {
	var Intent = {};
	
	Intent.parseIntentData = function(input) {
		var result = {};
		var dataList = input.split("\n");
		var domain;
		for (var i = 0; i < dataList.length; ++i ) {
			var text = dataList[i].trim();
			if (text == '') continue;

			if (text.charAt(0) == '#') {
				domain = text.replace('#', '');
				result[domain] = { 
					name: domain, 
					items: {}
				};
			} else if (domain != null) {
				result[domain]['items']['#' + domain + '%' + text] = { name: text };
			}
		}

		return result;
	}

	return Intent;
})();

var Utterance = (function() {
	var Utterance = {};

	Utterance.importFromCSV = function(csv) {
		var result = [];
		var raw, tagged;
		
		var items = csv.split("\n");
		for (var i = 0; i < items.length; ++i ) {
		var item = items[i].trim();
		if (item == '') continue;

		var texts = item.split(',');
			result.push({
				'raw' : texts[0],
				'tagged' : texts[1]
			})
		}
		
		return result;
	}
	
	Utterance.exportToCSV = function(input) {
		var result = '';
		for (var i = 0; i < input.length; ++i) {
			result += input[i]['raw'] + ',' + input[i]['tagged'] + '\n';
		}
		
		return result;
	}
		
	return Utterance;
	
}());
