package com.Fintech.OnlineBanking.token;
import com.Fintech.OnlineBanking.user.User;

import jakarta.persistence.Entity;
import jakarta.persistence.EnumType;
import jakarta.persistence.Enumerated;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.ManyToOne;
import jakarta.persistence.Table;

@Entity
@Table(name="")
public class Token {
	@Id 
	@GeneratedValue(strategy = GenerationType.IDENTITY)
 private Long id;
	private String token;
	@Enumerated(EnumType.STRING)
	private TokenType TokenType;
	private  boolean expired;
	private  boolean revoked;  
	@ManyToOne
	@JoinColumn(name="user_id")
	private User user;
	
	public User getUser() {
		return user;
	}
	public void setUser(User user) {
		this.user = user;
	}
	public Long getId() {
		return id;
	}
	public void setId(Long id) {
		this.id = id;
	}
	public String getToken() {
		return token;
	}
	public void setToken(String token) {
		this.token = token;
	}
	public TokenType getTokenType() {
		return TokenType;
	}
	public void setTokenType(TokenType tokenType) {
		TokenType = tokenType;
	}
	public boolean isExpired() {
		return expired;
	}
	public void setExpired(boolean expired) {
		this.expired = expired;
	}
	public boolean isRevoked() {
		return revoked;
	}
	public void setRevoked(boolean revoked) {
		this.revoked = revoked;
	}
	
	

}
