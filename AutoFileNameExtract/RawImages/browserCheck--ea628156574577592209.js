"use strict";
(self["webpackChunkolympus_frontend"] = self["webpackChunkolympus_frontend"] || []).push([[447],{

/***/ 51650:
/***/ (function(__unused_webpack_module, __unused_webpack___webpack_exports__, __webpack_require__) {

/* harmony import */ var core_js_modules_es_regexp_exec_js__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(47551);
/* harmony import */ var core_js_modules_es_regexp_exec_js__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(core_js_modules_es_regexp_exec_js__WEBPACK_IMPORTED_MODULE_0__);


/******/
(function () {
  // webpackBootstrap

  /******/
  "use strict";

  var __webpack_exports__ = {};
  ; // CONCATENATED MODULE: ./src/utils/supportedBrowsers.js

  /* eslint-disable */
  // GENERATED FILE DO NOT EDIT

  /* harmony default export */

  var supportedBrowsers = /((CPU[ +]OS|iPhone[ +]OS|CPU[ +]iPhone|CPU IPhone OS)[ +]+(11[_.]0|11[_.]([1-9]|\d{2,})|11[_.]2|11[_.]([3-9]|\d{2,})|(1[2-9]|[2-9]\d|\d{3,})[_.]\d+|12[_.]0|12[_.]([1-9]|\d{2,})|12[_.]5|12[_.]([6-9]|\d{2,})|(1[3-9]|[2-9]\d|\d{3,})[_.]\d+|13[_.]0|13[_.]([1-9]|\d{2,})|13[_.]7|13[_.]([8-9]|\d{2,})|(1[4-9]|[2-9]\d|\d{3,})[_.]\d+|14[_.]0|14[_.]([1-9]|\d{2,})|14[_.]4|14[_.]([5-9]|\d{2,})|14[_.]8|14[_.](9|\d{2,})|(1[5-9]|[2-9]\d|\d{3,})[_.]\d+|15[_.]0|15[_.]([1-9]|\d{2,})|(1[6-9]|[2-9]\d|\d{3,})[_.]\d+)(?:[_.]\d+)?)|(CFNetwork\/8.* Darwin\/17\.0\.\d+)|(CFNetwork\/8.* Darwin\/17\.2\.\d+)|(CFNetwork\/8.* Darwin\/17\.3\.\d+)|(CFNetwork\/8.* Darwin\/17\.\d+)|(Edge\/(94(?:\.0)?|94(?:\.([1-9]|\d{2,}))?|(9[5-9]|\d{3,})(?:\.\d+)?))|((Chromium|Chrome)\/(73\.0|73\.([1-9]|\d{2,})|(7[4-9]|[8-9]\d|\d{3,})\.\d+|83\.0|83\.([1-9]|\d{2,})|(8[4-9]|9\d|\d{3,})\.\d+)(?:\.\d+)?)|(Version\/(11\.0|11\.([1-9]|\d{2,})|(1[2-9]|[2-9]\d|\d{3,})\.\d+|12\.0|12\.([1-9]|\d{2,})|(1[3-9]|[2-9]\d|\d{3,})\.\d+|13\.0|13\.([1-9]|\d{2,})|(1[4-9]|[2-9]\d|\d{3,})\.\d+|14\.0|14\.([1-9]|\d{2,})|(1[5-9]|[2-9]\d|\d{3,})\.\d+|15\.0|15\.([1-9]|\d{2,})|(1[6-9]|[2-9]\d|\d{3,})\.\d+)(?:\.\d+)? Safari\/)|(Firefox\/(93\.0|93\.([1-9]|\d{2,})|(9[4-9]|\d{3,})\.\d+)\.\d+)|(Firefox\/(93\.0|93\.([1-9]|\d{2,})|(9[4-9]|\d{3,})\.\d+)(pre|[ab]\d+[a-z]*)?)/;
  ; // CONCATENATED MODULE: ./src/utils/browserInfo.ts

  /**
   * Heads up! This file makes up the bulk of the browserCheck.js entry file
   * Which, is compiled to very old Javascript to ensure that the code testing if
   * the browser is too old isn't too new for bad browsers. Practically this means
   * this file should:
   *
   * - NOT import other code in src (except the support regex)
   * - NOT use methods that would require polyfills in <= ES5 browsers
   */
  // @ts-ignore
  // eslint-disable-next-line import/extensions
  // Code based on https://github.com/arasatasaygin/is.js/blob/master/is.js.

  function getUserAgent() {
    var _window$navigator;

    return ((_window$navigator = window.navigator) == null ? void 0 : _window$navigator.userAgent) || '';
  }

  function isIos() {
    var userAgent = getUserAgent();
    return userAgent.toLowerCase().match(/(ipad|iphone).+?os (\d+)/) !== null;
  }

  function isAndroid() {
    var userAgent = getUserAgent();
    return userAgent.toLowerCase().match(/(android)/) !== null;
  } // Whitelist valid user agents that can login by directly passing
  // an access_token hash to the URL.


  var APP_USER_AGENT_REGEX = /ButterflyIQ/i;
  /**
   * The mobile app opens the education portal within an WebView and
   * applies custom user agent. Design differs slightly between a regular
   * mobile user and a user within the app so we need to differentiate.
   */

  function isInApp() {
    var userAgent = getUserAgent();
    return APP_USER_AGENT_REGEX.test(userAgent);
  }

  function isSupported() {
    var userAgent = getUserAgent();
    return supportedBrowsers.test(userAgent) || isInApp();
  }

  ; // CONCATENATED MODULE: ./src/browserCheck.js

  /* eslint-disable global-require */

  /**
   * WATCH OUT!!!
   *
   * This code is run in old browsers and cannot use anything that needs
   * polyfilling.
   *
   * Defensive try/catch in case of something too new slipping in accidentally.
   */

  try {
    // Not worth including the style infra for just this.
    var isSmallDevice = Math.min(window.screen.width, window.screen.height) < 768;

    if (!isSupported()) {
      window.location.replace(isSmallDevice ? "/unsupported-browser-mobile.html" : '/unsupported-browser.html');
    }
  } catch (err) {
    window.location.replace("/unsupported-browser.html");
  }
  /******/

})();

/***/ })

},
/******/ function(__webpack_require__) { // webpackRuntimeModules
/******/ var __webpack_exec__ = function(moduleId) { return __webpack_require__(__webpack_require__.s = moduleId); }
/******/ __webpack_require__.O(0, [551], function() { return __webpack_exec__(51650); });
/******/ var __webpack_exports__ = __webpack_require__.O();
/******/ }
]);
//# sourceMappingURL=browserCheck--ea628156574577592209.js.map