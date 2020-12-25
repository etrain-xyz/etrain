function changeLanguage(lang, link) {
	if (localStorage)
		localStorage.setItem('lang', lang);
	if (link)
		window.location.href = link;
}

function hideLoading() {
	document.querySelector('#outer').classList.add("hide");
}