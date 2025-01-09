<?php
$servername = "localhost";
$username = "root";
$password = "";
$dbname = "car_rental";

// Enable error reporting for debugging
ini_set('display_errors', 1);
error_reporting(E_ALL);

// Create connection
$conn = new mysqli($servername, $username, $password, $dbname);

// Check connection
if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}

// Set the charset for proper handling of special characters
if (!$conn->set_charset("utf8mb4")) {
    die("Error loading character set utf8mb4: " . $conn->error);
}

?>
