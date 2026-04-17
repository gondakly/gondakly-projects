package Gui;

import javax.swing.*;
import java.awt.event.*;
import dbutil.DBConnection;
import lms.Account;

public class AccountSignUpGUI extends JFrame {
    // GUI components
    private JTextField userNameField;
    private JPasswordField passwordField;
    private JButton signUpButton;
    private JButton backButton; // New Back button
    private DBConnection dbConnection; // Instance of DBConnection

    public AccountSignUpGUI() {
        // Initialize DBConnection
        dbConnection = new DBConnection();

        // Initialize GUI components
        userNameField = new JTextField(10);
        passwordField = new JPasswordField(10);
        signUpButton = new JButton("Sign Up");
        backButton = new JButton("Back"); // Initialize Back button

        // Add action listener to the sign up button
        signUpButton.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                // Get input values from fields
                String userName = userNameField.getText();
                // Retrieve password as char array
                char[] passwordChars = passwordField.getPassword();
                // Convert char array to string
                String password = new String(passwordChars);

                // Create an Account object with only username and password
                Account newAccount = new Account(userName, password);

                // Call the addAccount() method from DBConnection with the new account
                boolean success = dbConnection.addAccount(newAccount);

                // Show message based on success
                if (success) {
                    JOptionPane.showMessageDialog(null, "Account created successfully!");
                } else {
                    JOptionPane.showMessageDialog(null, "Failed to create account!");
                }
            }
        });

        // Add action listener to the back button
        backButton.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                // Close the current window (go back)
                dispose();
            }
        });

        // Create layout and add components
        JPanel panel = new JPanel();
        panel.add(new JLabel("Username:"));
        panel.add(userNameField);
        panel.add(new JLabel("Password:"));
        panel.add(passwordField);
        panel.add(signUpButton);
        panel.add(backButton); // Add Back button to the panel

        // Add panel to the frame
        this.add(panel);

        // Set frame properties
        this.setTitle("Account Sign Up");
        this.setSize(300, 200);
        this.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    }

    public void start() {
        // Set frame visible when starting the GUI
        this.setVisible(true);
    }
}
