<?php
// Database connection details
$servername = "localhost";
$username = "root"; // Replace with your database username
$password = ""; // Replace with your database password
$dbname = "car_rental";

// Create connection
$conn = new mysqli($servername, $username, $password, $dbname);

// Check connection
if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}

// Get form data
if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $pickupAddress = $_POST['pickupAddress'];
    $pickupDate = $_POST['pickupDate'];
    $pickupStartHour = $_POST['pickupStartHour'];
    $pickupEndHour = $_POST['pickupEndHour'];
    $carid = $_POST['CarID'];

    // Insert data into the Reservations table using prepared statements
    $sql = "INSERT INTO Reservations (pickup_address, pickup_date, pickup_start_hour, pickup_end_hour, CarID)
            VALUES (?, ?, ?, ?, ?)";

    $stmt = $conn->prepare($sql);
    $stmt->bind_param("ssssi", $pickupAddress, $pickupDate, $pickupStartHour, $pickupEndHour, $carid);

    if ($stmt->execute()) {
        echo "Reservation submitted successfully!";
        echo "<br>One Of Our Drivers Is On His Way";
    } else {
        echo "Error: " . $stmt->error;
    }

    // Close the statement
    $stmt->close();
}

// Close connection
$conn->close();
?>