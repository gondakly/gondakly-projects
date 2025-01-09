  // JavaScript to handle enabling/disabling the card number field
  const paymentMethod = document.getElementById('paymentMethod');
  const cardNumber = document.getElementById('cardNumber');

  paymentMethod.addEventListener('change', function () {
      if (paymentMethod.value === 'creditCard') {
          cardNumber.disabled = false; // Enable the card number field
      } else {
          cardNumber.disabled = true; // Disable the card number field
      }
  });