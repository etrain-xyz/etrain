if (localStorage && localStorage.getItem('lang')) {
	console.log(localStorage.getItem('lang'))
	setTimeout(function () {
		hideLoading();
	}, 1500)
} else {
	var requestUrl = window.location.protocol + '//ip-api.com/json';
	fetch(requestUrl).then(response => response.json())
		.then(json => {
			console.log('My country is: ' + json.countryCode);
			var en_url = window.location.origin + '/en';
			if (json.countryCode.toLowerCase() !== 'vn' && window.location.href.indexOf(en_url) < 0) {
				window.location.href = window.location.origin + '/en';
				changeLanguage("en");
			} else {
				changeLanguage("vi");
				hideLoading();
			}
		})
		.catch((error) => {
			console.error('Error:', error);
			changeLanguage('en');
			hideLoading();
		});
}