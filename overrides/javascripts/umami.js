function umami() {
  if (window.location.hostname !== "localhost" && window.location.hostname !== "127.0.0.1") {
    var script = document.createElement("script");
    script.defer = true;
    script.src = "https://umami.ricolxwz.io/script.js";
    script.setAttribute(
      "data-website-id",
      "83924929-2051-4bac-be81-bee6fbccf6c7"
    );
    document.head.appendChild(script);
  }
}
umami();