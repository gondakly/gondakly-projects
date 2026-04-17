package Gui;

import javax.swing.*;
import javax.swing.table.DefaultTableModel;
import java.sql.SQLException;
import java.util.ArrayList;
import dbutil.DBConnection;
import lms.Student;

public class DisplayStudents extends JFrame {
    private DBConnection dbConnection;
    private JTable studentTable;

    public DisplayStudents() {
        // Initialize DBConnection
        dbConnection = new DBConnection();

        // Set title
        setTitle("Display Students");

        // Create JTable to display students
        studentTable = new JTable();
        JScrollPane scrollPane = new JScrollPane(studentTable);

        // Add scroll pane to the frame
        add(scrollPane);

        // Set frame properties
        setSize(600, 400);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    }

    // Method to fetch and display students
    public void displayStudents() {
        // Get list of students from the database
        ArrayList<Student> students = dbConnection.displayAllStudents();

        // Create table model
        DefaultTableModel model = new DefaultTableModel();
        model.addColumn("Year");
        model.addColumn("ID");
        model.addColumn("Department");
        model.addColumn("Full Name");
        model.addColumn("Book");

        // Populate table model with student data
        for (Student student : students) {
            model.addRow(new Object[]{
                    student.getYear(),
                    student.getId(),
                    student.getDepartment(),
                    student.getFullName(),
                    student.getBook()
            });
        }

        // Set table model to the JTable
        studentTable.setModel(model);
    }

    public static void main(String[] args) {
        // Run the GUI on the event dispatch thread
        SwingUtilities.invokeLater(new Runnable() {
            public void run() {
                // Create an instance of DisplayStudents
                DisplayStudents displayStudents = new DisplayStudents();
                // Fetch and display students
                displayStudents.displayStudents();
                // Set the frame visible
                displayStudents.setVisible(true);
            }
        });
    }
}
