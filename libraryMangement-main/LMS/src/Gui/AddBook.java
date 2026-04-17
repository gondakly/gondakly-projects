package Gui;

import javax.swing.*;
import java.awt.event.*;
import dbutil.DBConnection;
import lms.Book;

public class AddBook extends JFrame {
    private JTextField bookIdField;
    private JTextField nameField;
    private JTextField authorField;
    private JTextField genreField;
    private JCheckBox availableCheckBox;
    private JTextField priceField;
    private JTextField pagesField;
    private JButton addButton;
    private JButton backButton; // New back button
    private DBConnection dbConnection;

    public AddBook() {
        // Initialize DBConnection
        dbConnection = new DBConnection();

        // Initialize GUI components
        setTitle("Add Book");
        bookIdField = new JTextField(10);
        nameField = new JTextField(10);
        authorField = new JTextField(10);
        genreField = new JTextField(10);
        availableCheckBox = new JCheckBox("Available");
        priceField = new JTextField(10);
        pagesField = new JTextField(10);
        addButton = new JButton("Add");
        backButton = new JButton("Back"); // Initialize back button

        // Add action listener to the add button
        addButton.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                addBook();
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
        panel.add(new JLabel("Book ID:"));
        panel.add(bookIdField);
        panel.add(new JLabel("Name:"));
        panel.add(nameField);
        panel.add(new JLabel("Author:"));
        panel.add(authorField);
        panel.add(new JLabel("Genre:"));
        panel.add(genreField);
        panel.add(availableCheckBox);
        panel.add(new JLabel("Price:"));
        panel.add(priceField);
        panel.add(new JLabel("Pages:"));
        panel.add(pagesField);
        panel.add(addButton);
        panel.add(backButton); // Add back button to the panel

        // Add panel to the frame
        add(panel);

        // Set frame properties
        setSize(400, 300);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    }

    private void addBook() {
        // Get input values from fields
        int bookId = Integer.parseInt(bookIdField.getText());
        String name = nameField.getText();
        String author = authorField.getText();
        String genre = genreField.getText();
        boolean available = availableCheckBox.isSelected();
        double price = Double.parseDouble(priceField.getText());
        int pages = Integer.parseInt(pagesField.getText());

        // Create a Book object
        Book newBook = new Book(bookId, name, author, genre, available, price, pages);

        // Call the addBook method from DBConnection with the new book
        boolean success = dbConnection.addBook(newBook);

        // Show message based on success
        if (success) {
            JOptionPane.showMessageDialog(null, "Book added successfully!");
        } else {
            JOptionPane.showMessageDialog(null, "Failed to add book!");
        }
    }

    public static void main(String[] args) {
        // Run the GUI on the event dispatch thread
        SwingUtilities.invokeLater(new Runnable() {
            public void run() {
                // Create an instance of AddBook and start it
                new AddBook().setVisible(true);
            }
        });
    }
}
