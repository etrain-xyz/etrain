function changeLanguage(lang, current_lang) {
	if (localStorage)
		localStorage.setItem('lang', lang);

	if (current_lang) {
		if (window.location.pathname == "/") {
			window.location.href = window.location.href
		} else {
			var pattern = {
				"vi": window.location.origin,
				"en": window.location.origin + "/en"
			};
			window.location.href = window.location.href.replace(pattern[current_lang], pattern[lang]);
		}
	} else {
		if (localStorage.getItem('lang') == "vi") {
			addAllClass('.post-en', 'hide');
			removeAllClass('.post-vi', 'hide');
		} else {
			addAllClass('.post-vi', 'hide');
			removeAllClass('.post-en', 'hide');
		}
	}
}

function removeAllClass(selector, class_name) {
	var list = document.querySelectorAll(selector);
	for (var i = 0; i < list.length; ++i) {
		list[i].classList.remove(class_name);
	}
}
function addAllClass(selector, class_name) {
	var list = document.querySelectorAll(selector);
	for (var i = 0; i < list.length; ++i) {
		list[i].classList.add(class_name);
	}
}

function hideLoading() {
	if (document.querySelector('#outer')) {
		document.querySelector('#outer').classList.add("hide");
	}	
}

function resizeWindow() {
	var new_w = 330;
	var new_h = 164;
	var margin_top = 3.75;
	if (window.innerWidth < 512) {
		new_w = 330 * window.innerWidth / 512;
		new_h = 164 * new_w / 330;
		delta = 512 - Math.round(window.innerWidth)
		if (window.innerWidth < 486) {
			delta = delta / 1.4;
		}
		mg = delta / 100
		margin_top = margin_top - mg
	}
	var divs = document.querySelectorAll('.screen-tv img');
	for (i = 0; i < divs.length; ++i) {
		var div = divs[i]
		div.setAttribute("width", Math.round(new_w));
		div.setAttribute("height", Math.round(new_h));
		div.parentElement.parentElement.style['margin-top'] = margin_top + 'rem';
	}
}

resizeWindow()

window.onresize = resizeWindow;