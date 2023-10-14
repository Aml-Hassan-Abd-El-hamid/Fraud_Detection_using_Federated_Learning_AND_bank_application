package com.Fintech.OnlineBanking.auth;

import java.io.IOException;

import java.net.http.HttpHeaders;
import java.util.Optional;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.authentication.BadCredentialsException;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.security.web.authentication.WebAuthenticationDetailsSource;
import org.springframework.stereotype.Service;
import org.springframework.http.HttpStatus;

import com.Fintech.OnlineBanking.config.JwtService;
import com.Fintech.OnlineBanking.domain.Message;
import com.Fintech.OnlineBanking.domain.UserInformation;
import com.Fintech.OnlineBanking.exceptions.UnauthorizedException;
import com.Fintech.OnlineBanking.repository.MessageRepository;
import com.Fintech.OnlineBanking.repository.UserInformationRespository;
import com.Fintech.OnlineBanking.service.SignUpService;
import com.Fintech.OnlineBanking.token.Token;
import com.Fintech.OnlineBanking.token.TokenRepository;
import com.Fintech.OnlineBanking.token.TokenType;
import com.Fintech.OnlineBanking.user.Role;
import com.Fintech.OnlineBanking.user.User;
import com.Fintech.OnlineBanking.user.UserRepository;
import com.fasterxml.jackson.core.exc.StreamWriteException;
import com.fasterxml.jackson.databind.DatabindException;
import com.fasterxml.jackson.databind.ObjectMapper;

import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;

@Service
//@RequiredArgsConstructor
public class AuthenticationService {
	@Autowired
	private UserRepository repository;

	@Autowired
	private PasswordEncoder passwordEncoder;
	@Autowired
	private JwtService jwtService;
	@Autowired
	private AuthenticationManager authenticationManager;
	@Autowired
	private TokenRepository tokenRepo;
	@Autowired
	UserInformationRespository userInfoRepo;
	@Autowired
	UserRepository userRepo;
	@Autowired
	MessageRepository messageRepo;
	@Autowired
	private SignUpService userService;

	private void saveUserToken(User user, String jwtToken) {
		Token token = new Token();
		token.setToken(jwtToken);
		token.setUser(user);
		token.setExpired(false);
		token.setRevoked(false);
		token.setTokenType(TokenType.BEARER);
		tokenRepo.save(token);
	}

	private void revokeAllUserTokens(User user) {
		var validUserTokens = tokenRepo.findAllValidTokensByUser(user.getId());
		if (validUserTokens.isEmpty())
			return;
		validUserTokens.forEach(t -> {
			t.setExpired(true);
			t.setRevoked(true);
		});
		tokenRepo.saveAll(validUserTokens);
	}

	public ResponseEntity<?> authenticate(AuthenticationRequest request) {
		Optional<User> user3 = userRepo.findByUsername(request.getEmail());

		UserInformation userFromDB = userBlocked(user3.get().getUserInfo().getNationalId());

		try {
			if (userFromDB.getRemainingOpportunities() > 0) {

				authenticationManager.authenticate(
						new UsernamePasswordAuthenticationToken(request.getEmail(), request.getPassword()));
				var user = repository.findByUsername(request.getEmail()).orElseThrow();
				var jwtToken = jwtService.generateToken(user);
				var refreshToken = jwtService.generateRefreshToken(user);
				revokeAllUserTokens(user);
				saveUserToken(user, refreshToken);

				AuthenticationResponse authenticationResponse = new AuthenticationResponse();
				authenticationResponse.setAccessToken(jwtToken);
				authenticationResponse.setRefreshToken(refreshToken);
				return ResponseEntity.ok().body(authenticationResponse);

			} else {
				Message m1 = messageRepo.findByMessageNumber("13");
				return ResponseEntity.status(HttpStatus.UNAUTHORIZED).body(m1);

			}
		} catch (BadCredentialsException ex) {
			userFromDB.setRemainingOpportunities(userFromDB.getRemainingOpportunities() - 1);
			userInfoRepo.save(userFromDB);
			Message m1 = messageRepo.findByMessageNumber("4");
			return ResponseEntity.status(HttpStatus.UNAUTHORIZED).body(m1);
		}
	}

	UserInformation userBlocked(long nationalId) {
		UserInformation userInfo = new UserInformation();
		try {
			userInfo = userInfoRepo.findByNationalId(nationalId);
		} catch (Exception e) {
		}

		return userInfo;
	}

	public void refreshToken(HttpServletRequest request, HttpServletResponse response)
			throws StreamWriteException, DatabindException, IOException {
		final String authHeader = request.getHeader(org.springframework.http.HttpHeaders.AUTHORIZATION);
		final String refreshToken;
		final String userEmail;
		if (authHeader == null || !authHeader.startsWith("Bearer ")) {
			return;
		}
		refreshToken = authHeader.substring(7);
		userEmail = jwtService.extractUsername(refreshToken);

		if (userEmail != null) {
			var user = this.repository.findByUsername(userEmail).orElseThrow();
			var isTokenValid = isRefreshTokenNotRevoked(refreshToken);

			if (jwtService.isTokenValid(refreshToken, user) && isTokenValid) {
				var accessToken = jwtService.generateToken(user);
				AuthenticationResponse authResponse = new AuthenticationResponse();
				authResponse.setAccessToken(accessToken);
				authResponse.setRefreshToken(refreshToken);

				new ObjectMapper().writeValue(response.getOutputStream(), authResponse);
			} else {
				throw new UnauthorizedException("invalid Refresh Token");

			}
		}
	}

	public Boolean isRefreshTokenNotRevoked(String refreshToken) {
		var isTokenValid = tokenRepo.findByToken(refreshToken).map(t -> !t.isExpired() && !t.isRevoked()).orElse(false);
		return isTokenValid;

	}
}
