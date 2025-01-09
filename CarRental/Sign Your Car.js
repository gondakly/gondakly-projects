document.getElementById('carForm').addEventListener('submit', function (event) {
    const driverPhone = document.getElementById('driverPhone').value;
    const pricePerHour = document.getElementById('pricePerHour').value;

    // Validate phone number (example: must be 10 digits)
    if (!/^\d{8}$/.test(driverPhone)) {
        alert('Driver Phone Number must be 10 digits.');
        event.preventDefault();
    }

    // Validate price per hour (must be positive)
    if (pricePerHour <= 0) {
        alert('Price Per Hour must be greater than 0.');
        event.preventDefault();
    }
});