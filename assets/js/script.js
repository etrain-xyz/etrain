function changeLanguage(lang, current_lang) {
	if (localStorage)
		localStorage.setItem('lang', lang);
	if (current_lang) {
		var pattern = {
			"vi": window.location.origin,
			"en": window.location.origin + "/en"
		};
		window.location.href = window.location.href.replace(pattern[current_lang],pattern[lang]);
	}
}

function hideLoading() {
	document.querySelector('#outer').classList.add("hide");
}