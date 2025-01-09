<?php
require 'db_connection.php';

session_start();

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $username = strtolower(trim($_POST['Username']));
    $password = trim($_POST['Password']);
    $role = trim($_POST['Role']);

    // Prepare SQL to fetch user details
    $sql = "SELECT Password, UserType, Username FROM Login WHERE Email = ? OR Username = ?";
    $stmt = $conn->prepare($sql);
    $stmt->bind_param("ss", $username, $username);
    $stmt->execute();
    $stmt->store_result();

    if ($stmt->num_rows > 0) {
        $stmt->bind_result($hashed_password, $usertype, $username_in_db);
        $stmt->fetch();

        // Check if the password is hashed or plaintext
        if (password_needs_rehash($hashed_password, PASSWORD_DEFAULT) || $hashed_password === $password) {
            // If the password is plaintext or needs rehashing, hash it and update the database
            $new_hashed_password = password_hash($password, PASSWORD_DEFAULT);

            $update_sql = "UPDATE Login SET Password = ? WHERE Username = ?";
            $update_stmt = $conn->prepare($update_sql);
            $update_stmt->bind_param("ss", $new_hashed_password, $username_in_db);
            $update_stmt->execute();

            // Update the hashed password variable for verification
            $hashed_password = $new_hashed_password;
        }

        // Verify the password
        if (password_verify($password, $hashed_password)) {
            if (strcasecmp($role, $usertype) === 0) {
                // Set session variables
                $_SESSION['logged_in'] = true;
                $_SESSION['user_role'] = $usertype;
                $_SESSION['username'] = $username_in_db;

                // Redirect based on role
                if ($usertype === 'Admin') {
                    header("Location: admin.php");
                } elseif ($usertype === 'Customer') {
                    header("Location: User Dashboard.php");
                }
                exit();
            } else {
                echo "Role does not match.";
            }
        } else {
            echo "Incorrect password.";
        }
    } else {
        echo "User not found.";
    }

    $stmt->close();
}
$conn->close();
?>