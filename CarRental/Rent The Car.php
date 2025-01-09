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
$pickupAddress = $_POST['pickupAddress'];
$pickupDate = $_POST['pickupDate'];
$pickupStartHour = $_POST['pickupStartHour'];
$pickupEndHour = $_POST['pickupEndHour'];




// Insert data into the Reservations table
$sql = "INSERT INTO Reservations ( pickup_address, pickup_date, pickup_start_hour, pickup_end_hour)
        VALUES ('$pickupAddress', '$pickupDate', '$pickupStartHour', '$pickupEndHour')";

if ($conn->query($sql) === TRUE) {
    echo "Reservation submitted successfully!";
    echo "<br>One Of Our Drivers Is On His Way";
} else {
    echo "Error: " . $sql . "<br>" . $conn->error;
}

// Close connection
$conn->close();
?>