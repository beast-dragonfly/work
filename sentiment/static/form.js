const review = document.querySelector("#review");
const counter = document.querySelector("#review-length");
const example = document.querySelector("#example");
const maxLength = Number(counter.dataset.maxLength);
const formatNumber = new Intl.NumberFormat("ru-RU");

function updateLength() {
    const length = Array.from(review.value).length;
    counter.textContent = `${formatNumber.format(length)} / ${formatNumber.format(maxLength)}`;
    review.setCustomValidity(length > maxLength ? `Максимум ${maxLength} символов.` : "");
}

review.addEventListener("input", updateLength);
example.addEventListener("click", () => {
    review.value = "Размер подошёл, но ткань тоньше, чем ожидала. В целом нормально.";
    updateLength();
    review.focus();
});

example.hidden = false;
updateLength();
