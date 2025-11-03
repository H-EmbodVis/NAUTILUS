document.addEventListener('DOMContentLoaded', () => {

    // Initialize the hero carousel
    bulmaCarousel.attach('#results-carousel', {
        slidesToScroll: 1,
        slidesToShow: 1,
        autoplay: true,
        loop: true,
        autoplaySpeed: 4000,
        duration: 800
    });

    // Initialize all sliders
    const sliders = document.querySelectorAll('.bulma-slider');
    sliders.forEach(slider => {
        bulmaSlider.attach(slider, {
            // Slider options can go here if needed
        });
    });

    // Get the element where the prompt will be displayed
    const promptDisplay = document.getElementById('prompt-display');
    const defaultPromptText = promptDisplay.querySelector('p').innerHTML;

    // Get all the comparison rows
    const comparisonRows = document.querySelectorAll('.comparison-row');

    // Add event listeners to each comparison row
    comparisonRows.forEach(row => {
        row.addEventListener('mouseover', () => {
            const promptText = row.getAttribute('data-prompt');
            promptDisplay.querySelector('p').innerHTML = `<strong>Prompt:</strong> ${promptText}`;
        });

        row.addEventListener('mouseout', () => {
            promptDisplay.querySelector('p').innerHTML = defaultPromptText;
        });
    });
});