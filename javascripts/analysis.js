// function umami() {
//   if (window.location.hostname !== "localhost" && window.location.hostname !== "127.0.0.1") {
//     var script = document.createElement("script");
//     script.defer = true;
//     script.src = "https://umami.ricolxwz.io/script.js";
//     script.setAttribute(
//       "data-website-id",
//       "3a5faee0-96f2-4bf6-b74e-ab3ae685794a"
//     );
//     document.head.appendChild(script);
//   }
// }
// umami();

function loadGoogleAnalytics() {
  if (window.location.hostname !== "localhost" && window.location.hostname !== "127.0.0.1") {
    var script = document.createElement("script");
    script.async = true;
    script.src = "https://www.googletagmanager.com/gtag/js?id=G-65D8M5V1CL";
    document.head.appendChild(script);
    window.dataLayer = window.dataLayer || [];
    function gtag() {
      dataLayer.push(arguments);
    }
    gtag("js", new Date());
    gtag("config", "G-65D8M5V1CL");
  }
}
loadGoogleAnalytics();