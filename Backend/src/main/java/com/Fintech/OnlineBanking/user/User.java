package com.Fintech.OnlineBanking.user;

import java.util.Collection;

import java.util.Date;
import java.util.List;

import org.springframework.security.core.GrantedAuthority;
import org.springframework.security.core.authority.SimpleGrantedAuthority;
import org.springframework.security.core.userdetails.UserDetails;

import com.Fintech.OnlineBanking.domain.UserInformation;
import com.Fintech.OnlineBanking.token.Token;

import jakarta.persistence.Entity;
import jakarta.persistence.EnumType;
import jakarta.persistence.Enumerated;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.OneToMany;
import jakarta.persistence.OneToOne;
import jakarta.persistence.Table;

@Entity
@Table(name="_user")
public class User implements UserDetails{
	@Id 
	@GeneratedValue(strategy = GenerationType.IDENTITY)
 private Long id;
	private String firstname;
	private String lastname;

	private String password;
	@Enumerated(EnumType.STRING)
	private Role role;
 
	@OneToMany(mappedBy="user")
	private List<Token>tokens;
	
	@OneToOne
	@JoinColumn(referencedColumnName = "nationalId")
	private UserInformation userInfo;
		private long remainingOpportunities=3;
	    private boolean locked = true;
	    private Date passwordChangeTime;
		private String username;
	
	
	public UserInformation getUserInfo() {
			return userInfo;
		}

		public void setUserInfo(UserInformation userInfo) {
			this.userInfo = userInfo;
		}

		public long getRemainingOpportunities() {
			return remainingOpportunities;
		}

		public void setRemainingOpportunities(long remainingOpportunities) {
			this.remainingOpportunities = remainingOpportunities;
		}

		public boolean isLocked() {
			return locked;
		}

		public void setLocked(boolean locked) {
			this.locked = locked;
		}

		public Date getPasswordChangeTime() {
			return passwordChangeTime;
		}

		public void setPasswordChangeTime(Date passwordChangeTime) {
			this.passwordChangeTime = passwordChangeTime;
		}

		public void setUsername(String username) {
			this.username = username;
		}

	public List<Token> getTokens() {
		return tokens;
	}

	public void setTokens(List<Token> tokens) {
		this.tokens = tokens;
	}

	public Long getId() {
		return id;
	}

	public void setId(Long id) {
		this.id = id;
	}

	public String getFirstname() {
		return firstname;
	}

	public void setFirstname(String firstname) {
		this.firstname = firstname;
	}

	public String getLastname() {
		return lastname;
	}

	public void setLastname(String lastname) {
		this.lastname = lastname;
	}


	public Role getRole() {
		return role;
	}

	public void setRole(Role role) {
		this.role = role;
	}

	public void setPassword(String password) {
		this.password = password;
	}


	@Override
	public String getPassword() {
		return password;
	}

	@Override
	public String getUsername() {
		return username;	}

	@Override
	public boolean isAccountNonExpired() {
		return true;
	}

	@Override
	public boolean isAccountNonLocked() {
		return true;
	}

	@Override
	public boolean isCredentialsNonExpired() {
		return true;
	}

	@Override
	public boolean isEnabled() {
		return true;
	}

	@Override
	public Collection<? extends GrantedAuthority> getAuthorities() {
		return List.of(new SimpleGrantedAuthority(role.name()));
	}
	
}
