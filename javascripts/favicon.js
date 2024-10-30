function updateFavicon() {
    const lightIcon = 'https://cdn.jsdelivr.net/gh/sigmax0124/logo@master/favion-big-mc-212121-000000-1.svg';
    const darkIcon = 'https://cdn.jsdelivr.net/gh/sigmax0124/logo@master/favion-big-mc-000000-212121-1.svg';
    const currentTheme = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    const favicon = document.querySelector('link[rel="icon"]');
    favicon.href = currentTheme === 'dark' ? darkIcon : lightIcon;
  }

window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', updateFavicon);
updateFavicon();