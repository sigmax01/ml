function umami() {
  if (window.location.hostname !== "localhost" && window.location.hostname !== "127.0.0.1") {
    var script = document.createElement("script");
    script.defer = true;
    script.src = "https://umami.ricolxwz.io/script.js";
    script.setAttribute(
      "data-website-id",
      "c718d24c-0695-4b1a-bc57-30f0a0ea4cfb"
    );
    document.head.appendChild(script);
  }
}
umami();