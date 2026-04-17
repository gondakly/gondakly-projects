package Gui;


import java.awt.event.*;
import dbutil.DBConnection;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JPasswordField;
import javax.swing.JTextField;
import javax.swing.SwingUtilities;

public class LoginGui extends JFrame {
    private JTextField userNameField;
    private JPasswordField passwordField;
    private JButton loginButton;
    private DBConnection dbConnection;

    public LoginGui() {
        // Initialize DBConnection
        dbConnection = new DBConnection();

        // Initialize GUI components
        setTitle("Login");
        userNameField = new JTextField(10);
        passwordField = new JPasswordField(10);
        loginButton = new JButton("Login");

        // Add action listener to the login button
        loginButton.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                login();
            }
        });

        // Create layout and add components
        JPanel panel = new JPanel();
        panel.add(new JLabel("Username:"));
        panel.add(userNameField);
        panel.add(new JLabel("Password:"));
        panel.add(passwordField);
        panel.add(loginButton);

        // Add panel to the frame
        add(panel);

        // Set frame properties
        setSize(300, 200);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    }

    private void login() {
        // Get input values from fields
        String userName = userNameField.getText();
        // Retrieve password as char array
        char[] passwordChars = passwordField.getPassword();
        // Convert char array to string
        String password = new String(passwordChars);

        // Check login credentials
        boolean loginSuccessful = dbConnection.Login(userName, password);

        // Show message based on login result
        if (loginSuccessful) {
            JOptionPane.showMessageDialog(null, "Login successful!");
            // Navigate to home class
            dispose(); // Close the current window
            java.awt.EventQueue.invokeLater(new Runnable() {
                public void run() {
                    new home().setVisible(true);
                }
            });
        } else {
            JOptionPane.showMessageDialog(null, "Invalid username or password!");
        }
    }
    
    // Method to perform login
    public void performLogin() {
        login();
    }
    public void start() {
        // Set frame visible when starting the GUI
        this.setVisible(true);
    }
    public static void main(String[] args) {
        // Run the GUI on the event dispatch thread
        SwingUtilities.invokeLater(new Runnable() {
            public void run() {
                // Create an instance of LoginGui and start it
                new LoginGui().setVisible(true);
            }
        });
    }
}
