package Gui;

import javax.swing.*;
import java.awt.event.*;
import dbutil.DBConnection;
import lms.Student;

public class AddStudent extends JFrame {
    private JTextField idField;
    private JTextField yearField;
    private JTextField departmentField;
    private JTextField fullNameField;
    private JTextField bookField;
    private JButton addButton;
    private JButton backButton; // New back button
    private DBConnection dbConnection;

    public AddStudent() {
        // Initialize DBConnection
        dbConnection = new DBConnection();

        // Initialize GUI components
        setTitle("Add Student");
        idField = new JTextField(10);
        yearField = new JTextField(10);
        departmentField = new JTextField(10);
        fullNameField = new JTextField(10);
        bookField = new JTextField(10);
        addButton = new JButton("Add");
        backButton = new JButton("Back"); // Initialize back button

        // Add action listener to the add button
        addButton.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                addStudent();
            }
        });

        // Add action listener to the back button
        backButton.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                // Close current window and go back to previous window (if applicable)
                dispose();
            }
        });

        // Create layout and add components
        JPanel panel = new JPanel();
        panel.add(new JLabel("ID:"));
        panel.add(idField);
        panel.add(new JLabel("Year:"));
        panel.add(yearField);
        panel.add(new JLabel("Department:"));
        panel.add(departmentField);
        panel.add(new JLabel("Full Name:"));
        panel.add(fullNameField);
        panel.add(new JLabel("Book:"));
        panel.add(bookField);
        panel.add(addButton);
        panel.add(backButton); // Add back button to the panel

        // Add panel to the frame
        add(panel);

        // Set frame properties
        setSize(400, 300);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    }

    private void addStudent() {
        // Get input values from fields
        int id = Integer.parseInt(idField.getText());
        int year = Integer.parseInt(yearField.getText());
        String department = departmentField.getText();
        String fullName = fullNameField.getText();
        String book = bookField.getText();

        // Create a Student object
        Student newStudent = new Student(id, year, department, fullName, book);

        // Call the addStudent method from DBConnection with the new student
        boolean success = dbConnection.addStudent(newStudent);

        // Show message based on success
        if (success) {
            JOptionPane.showMessageDialog(null, "Student added successfully!");
        } else {
            JOptionPane.showMessageDialog(null, "Failed to add student!");
        }
    }

    public static void main(String[] args) {
        // Run the GUI on the event dispatch thread
        SwingUtilities.invokeLater(new Runnable() {
            public void run() {
                // Create an instance of AddStudent and start it
                new AddStudent().setVisible(true);
            }
        });
    }
}
