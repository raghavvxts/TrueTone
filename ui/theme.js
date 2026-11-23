// Switch in place: recordings, transcript edits and model results stay intact.
const themeButton = element.querySelector('#theme-toggle');
const requestedTheme = new URL(window.location.href).searchParams.get('__theme');
let savedTheme;
try { savedTheme = window.localStorage.getItem('truetone-theme'); } catch (_) {}
const validTheme = value => value === 'dark' || value === 'light';
let theme = validTheme(requestedTheme) ? requestedTheme : validTheme(savedTheme) ? savedTheme :
    window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';

function applyTheme(nextTheme) {
    theme = nextTheme;
    const dark = theme === 'dark';
    if (!dark) document.querySelectorAll('.dark').forEach(node => node.classList.remove('dark'));
    document.documentElement.classList.toggle('dark', dark);
    document.body.classList.toggle('dark', dark);
    document.documentElement.style.colorScheme = theme;
    themeButton.textContent = dark ? 'Light mode' : 'Dark mode';
    themeButton.setAttribute('aria-label', dark ? 'Switch to light mode' : 'Switch to dark mode');
    document.querySelector('meta[name="theme-color"]')?.setAttribute('content', dark ? '#000000' : '#f5f2e9');
}
applyTheme(theme);
themeButton.addEventListener('click', () => {
    applyTheme(theme === 'dark' ? 'light' : 'dark');
    try { window.localStorage.setItem('truetone-theme', theme); } catch (_) {}
    const url = new URL(window.location.href);
    url.searchParams.set('__theme', theme);
    window.history.replaceState(window.history.state, '', url);
});
