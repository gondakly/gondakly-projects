
package lms;

public class Book {
    private int bookID;
    private String author;
    private String name;
    private String genre;
    private double price ;
    private int pages ;
    private boolean available;
    

    public Book() {
    }
    

    public Book(String name) {
        this.author = name;
    }

    public Book(int bookID, String author, String genre, double price, boolean available) {
        this.bookID = bookID;
        this.author = author;
        this.genre = genre;
        this.price = price;
        this.available = available;
    }

    public Book(int bookID, String name,String author, String genre,boolean available, double price, int pages) {
        this.bookID = bookID;
        this.author = author;
        this.name = name;
        this.genre = genre;
        this.price = price;
        this.pages = pages;
        this.available = available;
    }

    

    public int getBookID() {
        return bookID;
    }

    public void setBookID(int bookID) {
        this.bookID = bookID;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    
    public String getAuthor() {
        return author;
    }

    public void setAuthor(String author) {
        this.author = author;
    }

    public String getGenre() {
        return genre;
    }

    public void setGenre(String genre) {
        this.genre = genre;
    }

   

    public double getPrice() {
        return price;
    }

    public void setPrice(double price) {
        this.price = price;
    }

    public boolean isAvailable() {
        
        return available;
    }

    public void setAvailable(boolean available) {
        this.available = available;
    }

    public int getPages() {
        return pages;
    }

    public void setPages(int pages) {
        this.pages = pages;
    }

    @Override
    public String toString() {
        return "Book{" + "bookID=" + bookID + ", author=" + author + ", name=" + name + ", genre=" + genre + ", price=" + price + ", pages=" + pages + ", available=" + available + '}';
    }

    
}
