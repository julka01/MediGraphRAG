export function showError(message) {
  const popup = document.getElementById('error-popup');
  document.getElementById('error-message').textContent = message;
  popup.classList.remove('hidden', 'is-success');
}

export function showSuccess(message) {
  const popup = document.getElementById('error-popup');
  document.getElementById('error-message').textContent = message;
  popup.classList.remove('hidden');
  popup.classList.add('is-success');
  setTimeout(() => popup.classList.add('hidden'), 3000);
}
