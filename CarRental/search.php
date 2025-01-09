<?php
// Include the database connection file
include 'db_connection.php';

// Handle form submission
if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $seat = $_POST['seat'];
    $color = $_POST['color'];
    $price_per_hour = $_POST['price_per_hour'];

    // SQL query to search for cars
    $sql = "
        SELECT c.CarID, c.driver_name, c.driver_phone, c.model, c.year, c.plate_id, c.price_per_hour, cs.Color, cs.Seat
        FROM Cars c
        JOIN CarSpecs cs ON c.CarID = cs.CarID
        WHERE cs.Seat = ? AND cs.Color = ? AND c.price_per_hour <= ?
        AND c.status = 'active'
    ";

    // Prepare and execute the query
    $stmt = $conn->prepare($sql);
    $stmt->bind_param("ssd", $seat, $color, $price_per_hour);
    $stmt->execute();
    $result = $stmt->get_result();
}
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Search Cars</title>
    <style>
        /* General reset for the body */
body {
  margin: 0;
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100vh;
  background-color: #f0f0f0;
  font-family: Arial, sans-serif;
}

/* Container for the form box */
.form-box {
  width: 100%;
  background-color: #fff;
  border-radius: 10px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  overflow: hidden; /* Ensures black top aligns perfectly */
}

/* Black top section */
.form-box .top-section {
  background-color: #000;
  height: 50px;
}
.form-box .top-section h1{
    color: #FFDF00;
    font-family: Niagara Engraved;
    font-size:40px;
}

/* Content of the form */
.form-box .form-content {
  padding: 20px;
}

/* Example form styling */
.form-box input,
.form-box button {
  display: block;
  width: 400px;
  margin-bottom: 15px;
  padding: 10px;
  border: 1px solid #ccc;
  border-radius: 5px;
  font-size: 14px;
}

.form-box button {
  background-color: #000;
  color: #fff;
  border: none;
  cursor: pointer;
  transition: background-color 0.3s;
  border-color: #FFDF00;
}

.form-box button:hover {
  background-color:#FFDF00;
}
        table {
            width: 100%;
            border-collapse: collapse;
        }
        table, th, td {
            border: 1px solid black;
        }
        th, td {
            padding: 8px;
            text-align: left;
        }
    </style>
</head>
<body>
<div class="form-box">
        <div class="top-section"><h1>Search the car</h1></div>
        <div class="form-content">
    
    <form method="POST" action="">
        <label for="seat">Seat:</label>
        <input type="text" id="seat" name="seat" required>
        <br>
        <label for="color">Color:</label>
        <input type="text" id="color" name="color" required>
        <br>
        <label for="price_per_hour">Max Price per Hour:</label>
        <input type="number" id="price_per_hour" name="price_per_hour" step="0.01" required>
        <br>
        <button type="submit">Search</button>
    </form>

    <?php
    // Display search results
    if ($_SERVER["REQUEST_METHOD"] == "POST" && isset($result)) {
        if ($result->num_rows > 0) {
            echo "<h2>Search Results:</h2>";
            echo "<table>";
            echo "<tr><th>Car ID</th><th>Driver Name</th><th>Driver Phone</th><th>Model</th><th>Year</th><th>Plate ID</th><th>Price per Hour</th><th>Color</th><th>Seat</th></tr>";
            while ($row = $result->fetch_assoc()) {
                echo "<tr>";
                echo "<td>" . $row['CarID'] . "</td>";
                echo "<td>" . $row['driver_name'] . "</td>";
                echo "<td>" . $row['driver_phone'] . "</td>";
                echo "<td>" . $row['model'] . "</td>";
                echo "<td>" . $row['year'] . "</td>";
                echo "<td>" . $row['plate_id'] . "</td>";
                echo "<td>" . $row['price_per_hour'] . "</td>";
                echo "<td>" . $row['Color'] . "</td>";
                echo "<td>" . $row['Seat'] . "</td>";
                echo "</tr>";
            }
            echo "</table>";
        } else {
            echo "<p>No cars found matching your criteria.</p>";
        }
    }
    ?>

</body>
</html>

<?php
// Close the connection at the end
$conn->close();
?>