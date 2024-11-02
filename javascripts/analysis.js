function umami() {
  if (window.location.hostname !== "localhost" && window.location.hostname !== "127.0.0.1") {
    var script = document.createElement("script");
    script.defer = true;
    script.src = "https://cloud.umami.is/script.js";
    script.setAttribute(
      "data-website-id",
      "b3480344-dfac-4bc4-8af7-b1a99141f689"
    );
    document.head.appendChild(script);
  }
}
umami();