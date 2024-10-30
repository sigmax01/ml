function umami() {
  if (window.location.hostname !== "localhost" && window.location.hostname !== "127.0.0.1") {
    var script = document.createElement("script");
    script.defer = true;
    script.src = "https://umami.ricolxwz.io/script.js";
    script.setAttribute(
      "data-website-id",
      "3a5faee0-96f2-4bf6-b74e-ab3ae685794a"
    );
    document.head.appendChild(script);
  }
}
umami();