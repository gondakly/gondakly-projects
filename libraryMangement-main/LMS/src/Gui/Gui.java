package Gui;

import javax.swing.*;

public class Gui {
    private Login_SignUp loginSignUp;
    private home homePage;

    public Gui() {
        loginSignUp = new Login_SignUp();
        homePage = new home();
    }

    public void start() {
        // Present options to the user
        Object[] options = {"Login", "Sign Up"};
        int choice = JOptionPane.showOptionDialog(null, "Choose an action", "Library Management System",
                JOptionPane.DEFAULT_OPTION, JOptionPane.QUESTION_MESSAGE, null, options, options[0]);

        // Depending on the choice, start the respective GUI
        if (choice == 0) {
            loginSignUp.setVisible(true); // Start the Login_SignUp GUI
        } else if (choice == 1) {
            loginSignUp.dispose(); // Dispose of the Login_SignUp frame
            homePage.setVisible(true); // Start the home GUI
        }
    }

    public static void main(String[] args) {
        // Create an instance of Gui and start the application
        SwingUtilities.invokeLater(new Runnable() {
            public void run() {
                Gui gui = new Gui();
                gui.start();
            }
        });
    }
}
