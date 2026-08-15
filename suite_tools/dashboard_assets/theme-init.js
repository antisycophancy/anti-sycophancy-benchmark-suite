(function () {
  try {
    const stored = window.localStorage.getItem('benchmarkDashboardTheme');
    const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    const theme = stored || (prefersDark ? 'dark' : 'light');
    document.documentElement.dataset.theme = theme;
    document.documentElement.classList.toggle('dark', theme === 'dark');
  } catch (error) {
    document.documentElement.dataset.theme = 'light';
  }
})();
