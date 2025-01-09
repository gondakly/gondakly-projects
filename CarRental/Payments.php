<?php
// Include the database connection file
include 'db_connection.php';

// Handle form submission
if ($_SERVER["REQUEST_METHOD"] == "POST") {
    // Retrieve form data
    $paymentMethod = $_POST['payment'];
    $cardNo = ($paymentMethod == 'Credit Card') ? $_POST['cardNo'] : null;

    // Retrieve ReservationID (you can get this from a session, form, or database)
    $reservationID = 1; // Replace with the actual ReservationID (e.g., from a session or form)

    // Insert payment into the database
    $sql = "
        INSERT INTO Payments (ReservationID, CardNo, PaymentMethod)
        VALUES (?, ?, ?)
    ";

    // Prepare and execute the statement
    $stmt = $conn->prepare($sql);
    $stmt->bind_param("iis", $reservationID, $cardNo, $paymentMethod);

    if ($stmt->execute()) {
        echo "Payment processed successfully!";
    } else {
        echo "Error processing payment: " . $stmt->error;
    }

    // Close the statement and connection
    $stmt->close();
    $conn->close();
}
?>