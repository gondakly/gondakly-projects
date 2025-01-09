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
$driverName = $_POST['driverName'];
$driverPhone = $_POST['driverPhone'];
$model = $_POST['model'];
$year = $_POST['year'];
$plateID = $_POST['plateID'];
$pricePerHour = $_POST['pricePerHour'];
$status = $_POST['status'];

// Insert data into the Cars table
$sql = "INSERT INTO Cars (driver_name, driver_phone, model, year, plate_id, price_per_hour, status)
        VALUES ('$driverName', '$driverPhone', '$model', '$year', '$plateID', '$pricePerHour', '$status')";

if ($conn->query($sql) === TRUE) {
    echo "Car added successfully!";
} else {
    echo "Error: " . $sql . "<br>" . $conn->error;
}

// Close connection
$conn->close();
?>