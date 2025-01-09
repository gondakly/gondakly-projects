<?php
// Start session
session_start();

// Database connection
$conn = new mysqli('localhost', 'root', '', 'car_rental');

// Check connection
if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}

// Process the signup form
if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    // Get the form data
    $first_name = trim($_POST['FirstName']);
    $last_name = trim($_POST['LastName']);
    $email = trim($_POST['Email']);
    $password = trim($_POST['Password']);

    // Validate the inputs
    if (empty($first_name) || empty($last_name) || empty($email) || empty($password)) {
        echo "All fields are required.";
        exit();
    }

    // Check if the email already exists
    $check_email = "SELECT Email FROM signup WHERE Email = ?";
    $stmt_check = $conn->prepare($check_email);
    $stmt_check->bind_param("s", $email);
    $stmt_check->execute();
    $stmt_check->store_result();

    if ($stmt_check->num_rows > 0) {
        echo "This email is already registered.";
        $stmt_check->close();
        exit();
    }
    $stmt_check->close();

    // Hash the password before storing it
    $hashed_password = password_hash($password, PASSWORD_BCRYPT);

    // Insert data into the 'signup' table
    $sql_signup = "INSERT INTO signup (first_Name, last_Name, Email, Password, UserType) 
                   VALUES (?, ?, ?, ?, 'Customer')"; // 'Customer' is the default user type
    $stmt_signup = $conn->prepare($sql_signup);

    if ($stmt_signup === false) {
        die("Error preparing signup statement: " . $conn->error);
    }

    $stmt_signup->bind_param("ssss", $first_name, $last_name, $email, $hashed_password);

    if ($stmt_signup->execute()) {
        // Insert data into the 'login' table
        $sql_login = "INSERT INTO login (Username, Password, UserType, Email) 
                      VALUES (?, ?, 'Customer', ?)";
        $stmt_login = $conn->prepare($sql_login);

        if ($stmt_login === false) {
            die("Error preparing login statement: " . $conn->error);
        }

        $stmt_login->bind_param("sss", $email, $hashed_password, $email);

        if ($stmt_login->execute()) {
            // Store user details in session
            $_SESSION['logged_in'] = true;
            $_SESSION['user_role'] = 'Customer';
            $_SESSION['username'] = "$first_name $last_name";

            // Redirect to the User Dashboard
            header("Location: User Dashboard.html"); // Updated to redirect to the User Dashboard
            exit();
        } else {
            echo "Error inserting into login table: " . $stmt_login->error;
        }

        $stmt_login->close();
    } else {
        echo "Error inserting into signup table: " . $stmt_signup->error;
    }

    $stmt_signup->close();
}

$conn->close();
?>